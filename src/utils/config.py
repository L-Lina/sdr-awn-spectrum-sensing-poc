"""Shared experiment configuration and argparse definition for the runners in experiments/."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ExperimentConfig:
    snr: float
    mod: str
    attack: str
    topk: int
    threshold_factor: float
    window_size: int
    min_region_len: int
    merge_gap: int
    burst_len: int
    output_dir: str
    dry_run: bool
    n_samples: int = 8192
    # None means "use window_size" (legacy behavior, unchanged for anyone who
    # never passes --sensing-window-size). Resolved in
    # src/utils/pipeline.py:run_dry_run_experiment via
    # resolve_sensing_window_size() below, NOT at config-construction time,
    # so direct-API callers who build ExperimentConfig by hand (not through
    # args_to_config/run_batch.py) still get correct behavior.
    sensing_window_size: Optional[int] = None
    use_real_topk: bool = False
    use_real_awn: bool = False
    checkpoint: str = "external/adversarial-rf/2016.10a_AWN.pkl"
    device: str = "cpu"
    attack_eps: float = 0.03
    use_real_attack: bool = False
    attack_temperature: float = 1.0
    attack_diagnostics: bool = False
    # Global reproducibility seed: seeds random/numpy/torch(+cuda) once at the
    # top of run_dry_run_experiment (src/utils/pipeline.py) AND is threaded
    # through to generate_synthetic_iq / dummy_awn_inference / dummy_attack /
    # AttackAdapter.apply's own seed= parameters, so every source of
    # randomness in one run uses the same value. Default 0 matches the prior
    # hardcoded SEED=0 in pipeline.py, so omitting --seed reproduces prior
    # behavior exactly.
    seed: int = 0
    # CW-only strength knobs (src/adapters/attack_adapter.py:_build_torchattacks).
    # fgsm/pgd never read these. attack_eps is NOT applicable to cw -- these
    # three parameters are deliberately separate, never derived from
    # attack_eps. Defaults match the previously hardcoded CW values, so
    # omitting --cw-c/--cw-steps/--cw-lr reproduces prior CW behavior exactly.
    cw_c: float = 1.0
    cw_steps: int = 20
    cw_lr: float = 0.01
    # RadioML (RML2016.10a) real-sample IQ source, as an alternative to the
    # synthetic generator -- see src/sensing/radioml_source.py. "synthetic"
    # (default) reproduces all prior behavior exactly and ignores every
    # dataset_*/sample_index/embed_snr_margin field below. "radioml" makes
    # dataset_path/dataset_mod/dataset_snr all REQUIRED (checked in
    # src/utils/pipeline.py, since validating dataset_mod/dataset_snr
    # against the actual pickle's available keys requires opening the
    # file) and BYPASSES generate_synthetic_iq entirely -- `snr`/`mod`
    # above are the synthetic generator's own inputs and are simply unused
    # in this mode, never silently reinterpreted as the RadioML ground
    # truth (that's what dataset_mod/dataset_snr are for).
    iq_source: str = "synthetic"
    dataset_path: Optional[str] = None
    dataset_mod: Optional[str] = None
    dataset_snr: Optional[int] = None
    sample_index: int = 0
    # How much the embedded RadioML burst's own power exceeds the
    # surrounding synthetic capture-noise floor (src/sensing/
    # radioml_source.py:embed_sample_in_noise) -- deliberately distinct
    # from the RadioML sample's own internal (mod,snr)-label SNR, which is
    # already baked into the loaded sample and not re-derivable from it.
    embed_snr_margin: float = 20.0
    # Multi-burst RadioML mode (src/sensing/radioml_source.py:
    # embed_multiple_samples_in_noise, src/sensing/ground_truth_metrics.py:
    # compute_multi_burst_sensing_metrics). num_bursts<=1 (the default)
    # takes the EXACT SAME single-burst code path as before (dataset_mod/
    # dataset_snr/sample_index above), completely unchanged -- this is what
    # guarantees the pre-existing single-burst behavior is not disturbed.
    # num_bursts>1 requires dataset_mod_list/dataset_snr_list/
    # sample_index_list to all be set, each a list of exactly num_bursts
    # entries (validated in validate_experiment_config); dataset_mod/
    # dataset_snr/sample_index (singular) are then unused.
    num_bursts: int = 1
    dataset_mod_list: Optional[List[str]] = None
    dataset_snr_list: Optional[List[int]] = None
    sample_index_list: Optional[List[int]] = None
    # Gap (in samples) drawn uniformly from [min_burst_gap, max_burst_gap]
    # before EACH burst (including a "leading gap" before the first one).
    # Setting min_burst_gap == max_burst_gap makes every gap an exact,
    # deterministic value -- used by the merge-gap main-pipeline test cases
    # (docs/parameter_validation.md section 15) to get precise truth-burst
    # spacing. Only meaningful when num_bursts > 1.
    min_burst_gap: int = 50
    max_burst_gap: int = 50
    # Optional exact, per-burst gap list (length == num_bursts), overriding
    # min_burst_gap/max_burst_gap's random sampling entirely -- needed when
    # different gaps are required at different positions in the same run
    # (see src/sensing/radioml_source.py:embed_multiple_samples_in_noise).
    burst_gap_list: Optional[List[int]] = None
    # Optional exact, per-burst amplitude multiplier (length == num_bursts,
    # each > 0), applied before the shared noise floor is computed --
    # needed to construct a genuinely low-energy burst on demand (real
    # RadioML samples have only modest inherent power differences across
    # mod/snr combos) for controlled detection-boundary test cases (see
    # src/sensing/radioml_source.py:embed_multiple_samples_in_noise).
    burst_power_scale_list: Optional[List[float]] = None
    # Segment-alignment policy (src/sensing/segmentation.py:select_aligned_segments,
    # docs/parameter_validation.md section 18/20). None (default) means "use
    # the source-aware default" -- resolved in src/utils/pipeline.py via
    # resolve_alignment_policy() below, NOT at config-construction time, same
    # pattern as sensing_window_size, so this applies uniformly regardless of
    # whether cfg came from argparse or was built directly. Section 20 sets
    # that resolution to "max-energy" for iq_source="radioml" (validated in
    # section 18.4/19.4 to recover oracle-path accuracy) and "naive" for
    # "synthetic" (unchanged -- section 18's diagnosis concerns real RadioML
    # samples specifically, no regression risk for synthetic). An explicitly
    # passed value (CLI --alignment-policy or direct ExperimentConfig(...)
    # construction) is NEVER overridden by the source-aware default.
    # "naive" reproduces every prior round's segment_regions() behavior
    # byte-for-byte -- fixed, non-overlapping seg_len windows starting at
    # each detected region's own start, which can be 53-61 samples before
    # the true burst start (energy_detect's smoothing widens the region's
    # leading edge), so a region with 100% region-level captured_signal_ratio
    # can still yield a segment that's only ~52-63% true-burst signal.
    # "max-energy" selects, per region, the single seg_len sliding window
    # (hop=segment_hop) with the highest mean power -- never references
    # true_burst_start/true_burst_end.
    alignment_policy: Optional[str] = None
    # Sliding-window step (samples) used by max-energy's candidate search
    # (and reported, informationally, as candidate_count even under naive).
    # Default 1 (every possible offset) for correctness testing; a batch run
    # over many combos may want a larger hop to reduce candidate-search cost.
    segment_hop: int = 1
    # AWN-input-boundary preprocessing policy (src/sensing/normalize.py:
    # apply_awn_preprocess, docs/parameter_validation.md section 19/20).
    # None (default) means "use the source-aware default", resolved in
    # src/utils/pipeline.py via resolve_awn_preprocess() below -- same
    # None-means-resolve-downstream pattern as alignment_policy above.
    # Section 20 sets that resolution to "radioml-native" for
    # iq_source="radioml" (validated in section 19.4 to recover
    # oracle-path accuracy) and "legacy-unit-power" for "synthetic"
    # (unchanged). An explicitly passed value is NEVER overridden.
    # "legacy-unit-power" -- normalize_segments()'s unit-average-power
    # rescale. "radioml-native" -- no rescaling at all, matching traced
    # evidence that external/adversarial-rf never normalizes between its
    # dataset loader and AWN.forward(). Applied ONLY at the AWN input
    # boundary in src/utils/pipeline.py -- never inside
    # segmentation.py/energy_detection.py.
    awn_preprocess: Optional[str] = None
    # Extended attack parameter surface (this round) -- covers every
    # attack beyond the original fgsm/pgd/cw set (see
    # src/adapters/attack_adapter.py:_ATTACK_ACCEPTED_PARAMS,
    # docs/ATTACK_NAME_MAPPING.md). Every field here defaults to None,
    # meaning "not explicitly set" -- AttackAdapter.apply()'s attack_params
    # dict only ever contains the non-None entries, so an attack whose
    # constructor doesn't accept a given field simply never sees it, and an
    # omitted field lets torchattacks' OWN installed default apply (never
    # duplicated/hardcoded here). fgsm/pgd/cw's pre-existing hardcoded
    # defaults (pgd alpha=eps/4 steps=10; cw from cw_c/cw_steps/cw_lr) are
    # completely unaffected when these are all left at None.
    attack_alpha: Optional[float] = None
    attack_steps: Optional[int] = None
    attack_random_start: Optional[bool] = None
    attack_decay: Optional[float] = None
    attack_resize_rate: Optional[float] = None
    attack_diversity_prob: Optional[float] = None
    attack_momentum_n: Optional[int] = None  # torchattacks' "N" kwarg (vmifgsm/vnifgsm)
    attack_beta: Optional[float] = None  # fab's step-size beta OR ead's L1-weight beta (mutually exclusive per attack)
    attack_overshoot: Optional[float] = None  # deepfool
    attack_kappa: Optional[float] = None  # cw/ead confidence margin
    attack_lr: Optional[float] = None  # ead's Adam learning rate (distinct from --cw-lr)
    attack_norm: Optional[str] = None  # fab/square/apgd/apgdt/autoattack
    attack_n_restarts: Optional[int] = None
    attack_loss: Optional[str] = None  # apgd
    attack_eot_iter: Optional[int] = None  # apgd/apgdt
    attack_rho: Optional[float] = None  # apgd/apgdt
    attack_alpha_max: Optional[float] = None  # fab
    attack_eta: Optional[float] = None  # fab
    attack_multi_targeted: Optional[bool] = None  # fab
    attack_n_classes: Optional[int] = None  # fab/apgdt/autoattack (default 11, AWN's real class count, if unset)
    attack_internal_seed: Optional[int] = None  # fab/square/apgd/apgdt/autoattack's own internal seed= kwarg
    attack_n_queries: Optional[int] = None  # square
    attack_p_init: Optional[float] = None  # square
    attack_resc_schedule: Optional[bool] = None  # square
    attack_version: Optional[str] = None  # autoattack
    attack_binary_search_steps: Optional[int] = None  # ead
    attack_max_iterations: Optional[int] = None  # ead
    attack_initial_const: Optional[float] = None  # ead
    attack_abort_early: Optional[bool] = None  # ead
    attack_ead_variant: Optional[str] = None  # "eadl1" (default) or "eaden"

    # --- Dataset / sample-selection surface (round: parameter acceptance
    # fixes) ---
    # Only one dataset is actually supported end-to-end (RML2016.10a, the
    # pinned checkpoint's own training dataset) -- an explicit field so a
    # typo'd/unsupported dataset name is rejected loudly instead of being
    # silently ignored (dataset_path's own file content was previously the
    # ONLY thing that mattered). "RML2016.10a" reproduces all prior
    # behavior exactly.
    dataset: str = "RML2016.10a"
    # mod_filter/snr_filter: NOT a batch-iteration driver (this remains a
    # single-combo tool -- see samples_per_cell below for why iterating
    # multiple combos per invocation was deliberately NOT built this
    # round) -- a WHITELIST/GUARD applied to the single selected
    # dataset_mod/dataset_snr. None (default, both) reproduces prior
    # behavior exactly (no filter applied). When set, dataset_mod must be
    # a member of mod_filter (checked against RML2016_10A_CLASSES) and
    # dataset_snr must be a member of snr_filter, or the run is rejected
    # before any sensing/AWN work starts.
    mod_filter: Optional[List[str]] = None
    snr_filter: Optional[List[int]] = None
    # samples_per_cell: distinct, deliberately NOT the same field as the
    # pre-existing n_samples (which means long synthetic-stream length in
    # samples, unrelated and unchanged) -- this is a per-(dataset_mod,
    # dataset_snr) sample_index BUDGET: when set, sample_index must be <
    # samples_per_cell, or the run is rejected. None (default) means no
    # such bound is enforced (only RML2016.10a's own per-cell block size,
    # 1000, bounds sample_index, exactly as before).
    samples_per_cell: Optional[int] = None
    # burst_insert_position: "random" (default, reproduces
    # embed_sample_in_noise's existing seeded-random placement exactly),
    # "center", or "explicit" (requires burst_insert_position_index).
    # RadioML single-burst mode only. See
    # src/sensing/radioml_source.py:embed_sample_in_noise_at_position.
    burst_insert_position: str = "random"
    burst_insert_position_index: Optional[int] = None
    # batch_size: how many segments AWNModelAdapter.infer() processes in a
    # single real forward-pass call, when the caller has more than one
    # segment available (multi-region sensing, or a caller-driven
    # multi-sample batch such as experiments/validate_pipeline_parameters.py's
    # smoke test) -- chunks a larger segment array into sub-batches of
    # this size rather than either one giant call or one-at-a-time calls.
    # run_dry_run_experiment's own single-combo call sites are usually
    # N=1 or N=(detected region count), both already <= any sane
    # batch_size, so this has no effect on their existing behavior unless
    # more segments are present than batch_size allows through at once.
    batch_size: int = 1
    # experiment_name: optional human-readable tag folded into the
    # manifest/output; sanitized (path separators and other illegal
    # filename characters stripped) since it is never used blindly as a
    # raw path component.
    experiment_name: Optional[str] = None
    # overwrite: default False -- if output_dir already contains a
    # summary.csv, run_dry_run_experiment now refuses to proceed unless
    # overwrite=True is set explicitly. Only ever touches the exact
    # output_dir given, never anything else under results/.
    overwrite: bool = False

    # --- Real IQ file input surface (this round, additive) ---
    # iq_source's third legal value: "cfile" -- loads a real (or any raw
    # binary) IQ file via src/io/iq_file_source.py:load_iq_file instead of
    # generating synthetic noise or reading a RadioML dataset sample.
    # "synthetic" and "radioml" are COMPLETELY UNCHANGED -- cfile mode
    # never creates a synthetic long stream and never reads a RadioML
    # sample; the reverse also holds (synthetic/radioml modes never touch
    # input_path or any iq_* field below). No ground truth (true_start/
    # true_end/oracle crop/detection_probability/captured_signal_ratio/
    # boundary_error) is available in cfile mode -- these stay None
    # throughout, exactly like the pre-existing "no radioml_meta" case
    # already handled by run_dry_run_experiment's ground_truth branch, not
    # a new code path. Optional true_label_mod lets a caller who DOES know
    # the ground-truth modulation for a captured file supply it (for
    # accuracy/attack-success metrics); omitting it leaves those metrics
    # None/unavailable, never fabricated as "unknown"==0 or similar.
    input_path: Optional[str] = None
    iq_format: str = "complex64"
    iq_endianness: str = "native"
    iq_scale: Optional[float] = None
    iq_sample_rate: Optional[float] = None
    iq_offset_samples: int = 0
    iq_max_samples: Optional[int] = None
    iq_channel_count: int = 1
    true_label_mod: Optional[str] = None

    # --- CPU thread configuration (this round, additive) ---
    # torch_num_threads: None (default) means "leave torch's own/environment
    # default thread count untouched" -- byte-for-byte the prior behavior,
    # since no code path called torch.set_num_threads() before this round
    # outside of standalone experiments/ benchmark scripts (which set and
    # restore it locally, never touching this config). A positive int calls
    # torch.set_num_threads(value) once, near the top of run_dry_run_experiment
    # (src/utils/pipeline.py), before any AWN/attack backend is constructed.
    # 0 and negative values are rejected in validate_experiment_config --
    # torch.set_num_threads(0) does not mean "unbounded", it is simply not a
    # meaningful thread count. This does not change Phase 0-4 default
    # behavior (all of them omit --torch-threads, which stays None).
    torch_num_threads: Optional[int] = None


def build_attack_params(cfg: "ExperimentConfig") -> Dict[str, object]:
    """Collects every non-None attack_* field on cfg into the flat dict
    AttackAdapter.apply(attack_params=...) expects, translating this
    config's CLI-facing names to the exact torchattacks constructor kwarg
    names (e.g. attack_momentum_n -> "N", attack_ead_variant ->
    "_ead_variant"). Only called from src/utils/pipeline.py; the many
    direct-API runner scripts (experiments/run_phase*.py) call
    AttackAdapter.apply() themselves and build their own (usually empty)
    attack_params dict, unaffected by this function."""
    mapping = {
        "attack_alpha": "alpha", "attack_steps": "steps", "attack_random_start": "random_start",
        "attack_decay": "decay", "attack_resize_rate": "resize_rate", "attack_diversity_prob": "diversity_prob",
        "attack_momentum_n": "N", "attack_beta": "beta", "attack_overshoot": "overshoot",
        "attack_kappa": "kappa", "attack_lr": "lr", "attack_norm": "norm", "attack_n_restarts": "n_restarts",
        "attack_loss": "loss", "attack_eot_iter": "eot_iter", "attack_rho": "rho",
        "attack_alpha_max": "alpha_max", "attack_eta": "eta", "attack_multi_targeted": "multi_targeted",
        "attack_n_classes": "n_classes", "attack_internal_seed": "seed", "attack_n_queries": "n_queries",
        "attack_p_init": "p_init", "attack_resc_schedule": "resc_schedule", "attack_version": "version",
        "attack_binary_search_steps": "binary_search_steps", "attack_max_iterations": "max_iterations",
        "attack_initial_const": "initial_const", "attack_abort_early": "abort_early",
        "attack_ead_variant": "_ead_variant",
    }
    return {tk: getattr(cfg, cf) for cf, tk in mapping.items() if getattr(cfg, cf) is not None}


# ---------------------------------------------------------------------------
# Reusable boundary validators. Each raises a plain ValueError with a message
# of the form "<name> must be ..., got <value>" -- used both by the argparse
# type= helpers below (CLI-time errors) and by validate_experiment_config()
# (called from src/utils/pipeline.py as the adapter/algorithm-boundary guard
# for direct-API callers who construct ExperimentConfig without going through
# argparse at all, e.g. experiments/run_batch.py's own ExperimentConfig(...)
# construction). merge_gap and topk are intentionally not covered by any of
# this yet -- see docs/parameter_validation.md for why.
# ---------------------------------------------------------------------------

def require_positive_finite_float(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value}")
    return value


def require_finite_float(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def require_nonneg_finite_float(name: str, value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value}")
    return value


def require_positive_int(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def require_nonneg_int(name: str, value: int) -> int:
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}")
    return value


def require_valid_topk(name: str, value) -> int:
    """Direct-API entry-point guard for topk (src/adapters/topk_adapter.py:
    TopKAdapter.apply(), src/adapters/defense_adapter.py:dummy_topk_defense).
    Deliberately does NOT restrict the *range* of topk -- topk<=0 keeps its
    existing bypass semantics (return input unchanged) and topk > the FFT
    bin count keeps its existing clamp semantics (min(topk, T)); both are
    unaffected by this function. This only rejects values that can never be
    a meaningful bin count at all: non-numeric, NaN/Inf, or a genuine
    fractional part (e.g. 1.5) -- previously such values were silently
    truncated via a bare int(topk) inside each backend, and NaN/Inf reached
    int() at all only inside the real/dummy backends themselves, sometimes
    after a real-backend failure had already triggered a fallback attempt
    (see docs/parameter_validation.md section 12.3 for the pre-fix
    behavior). Called BEFORE any backend selection, so a rejection here
    surfaces as an immediate ValueError and never gets a chance to trigger
    TopKAdapter's real-backend-failed-so-fall-back-to-dummy path. The
    --topk CLI flag itself is unaffected -- it already only ever produces
    plain ints via argparse's type=int, which always satisfies this check."""
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric, got {value!r}")
    if not math.isfinite(fvalue):
        raise ValueError(f"{name} must be finite (not NaN/Inf), got {value!r}")
    if fvalue != int(fvalue):
        raise ValueError(f"{name} must not have a fractional part, got {value!r}")
    return int(fvalue)


# RML2016.10a's actual SNR label set (dB) -- matches ALL_SNRS already used
# independently in every experiments/run_phase*.py script this session
# (e.g. run_phase1_sensing_baseline.py) and RML2016_10A_CLASSES's own
# 11-class ordering (src/sensing/radioml_source.py). Duplicated here as a
# plain constant (not imported from radioml_source.py, which would need to
# either open the dataset file or hardcode the same list itself) so
# mod_filter/snr_filter/dataset validation never needs to touch disk.
RML2016_10A_VALID_SNRS = list(range(-20, 20, 2))
RML2016_10A_MODULATIONS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
                            "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
SUPPORTED_DATASETS = ("RML2016.10a",)


def require_valid_topk_strict(name: str, value: int) -> int:
    """Strict [1, 128] Top-K range check -- deliberately SEPARATE from
    require_valid_topk() above, which src/adapters/topk_adapter.py:
    TopKAdapter.apply() (and src/adapters/defense_adapter.py) still call
    unmodified, preserving their existing, documented bypass (topk<=0) /
    clamp (topk>T) semantics for any direct-API caller. This function is
    for the FORMAL CLI/config entry point only (called from
    args_to_config(), not from validate_experiment_config() -- see that
    function's own docstring for why the shared validator does not call
    this): a formal, CLI-launched experiment must not silently accept an
    out-of-range K and either no-op or clamp it without telling the
    caller. 128 is the fixed upper bound because this pipeline's segment
    length (window_size / AWN input T) is always 128 samples in every
    formal round to date; this is not a general "must equal window_size"
    rule, just this project's own fixed value."""
    ivalue = require_valid_topk(name, value)
    if not (1 <= ivalue <= 128):
        raise ValueError(f"{name} must satisfy 1 <= {name} <= 128 for a formal CLI-launched run, got {ivalue}")
    return ivalue


def require_valid_min_region_len_strict(name: str, value: int) -> int:
    """Strict min_region_len > 0 check -- deliberately SEPARATE from the
    require_nonneg_int("min_region_len", ...) check inside
    validate_experiment_config() below, which allows 0 (a real,
    documented, deliberate value meaning "no minimum region-length
    filter" -- used as the FIXED default in every formal round this
    session: Phase 0-4, the four-path Spectrum Sensing Utility
    Experiment, and attack-compatibility all use min_region_len=0, most
    of them via DIRECT calls to filter_by_min_length/select_aligned_segments
    that never go through this function at all, but
    experiments/run_phase1_sensing_baseline.py specifically calls
    run_dry_run_experiment(cfg) directly with min_region_len=0, which DOES
    go through validate_experiment_config() -- changing that shared
    function to reject 0 would make Phase 1's own formal run
    unreproducible through this exact code path. This stricter function
    is therefore called ONLY from args_to_config() (the CLI entry point),
    not from validate_experiment_config(), so a NEW CLI-launched run must
    supply a positive min_region_len while Phase 1's already-completed,
    documented direct-API invocation remains exactly reproducible."""
    ivalue = require_nonneg_int(name, value)
    if ivalue <= 0:
        raise ValueError(
            f"{name} must be a positive integer for a formal CLI-launched run (got {ivalue}); "
            f"0 ('no minimum length filter') is only accepted via direct ExperimentConfig/"
            f"validate_experiment_config construction, matching Phase 1's existing formal usage."
        )
    return ivalue


def validate_experiment_config(cfg: ExperimentConfig) -> None:
    """Boundary validation for direct-API callers of run_dry_run_experiment(cfg)
    that bypass argparse entirely. Covers exactly the parameters with a
    finalized rule as of this round; merge_gap and topk are out of scope."""
    require_positive_finite_float("threshold_factor", cfg.threshold_factor)
    require_positive_int("window_size", cfg.window_size)
    if cfg.sensing_window_size is not None:
        require_positive_int("sensing_window_size", cfg.sensing_window_size)
    require_nonneg_int("min_region_len", cfg.min_region_len)
    require_positive_int("burst_len", cfg.burst_len)
    require_finite_float("snr_db", cfg.snr)
    require_nonneg_finite_float("attack_eps", cfg.attack_eps)
    require_positive_finite_float("attack_temperature", cfg.attack_temperature)
    require_positive_finite_float("cw_c", cfg.cw_c)
    require_positive_int("cw_steps", cfg.cw_steps)
    require_positive_finite_float("cw_lr", cfg.cw_lr)
    if cfg.iq_source not in ("synthetic", "radioml", "cfile"):
        raise ValueError(f"iq_source must be 'synthetic', 'radioml', or 'cfile', got {cfg.iq_source!r}")
    if cfg.iq_source == "cfile":
        # Additive this round -- entirely separate from the synthetic/
        # radioml branches above/below, which never read any of these
        # fields. See src/io/iq_file_source.py for the loader itself.
        from src.io.iq_file_source import SUPPORTED_ENDIANNESS, SUPPORTED_IQ_FORMATS
        if cfg.input_path is None:
            raise ValueError("--iq-source cfile requires input_path to be set")
        if cfg.iq_format not in SUPPORTED_IQ_FORMATS:
            raise ValueError(f"iq_format must be one of {SUPPORTED_IQ_FORMATS}, got {cfg.iq_format!r}")
        if cfg.iq_endianness not in SUPPORTED_ENDIANNESS:
            raise ValueError(f"iq_endianness must be one of {SUPPORTED_ENDIANNESS}, got {cfg.iq_endianness!r}")
        require_nonneg_int("iq_offset_samples", cfg.iq_offset_samples)
        if cfg.iq_max_samples is not None:
            require_positive_int("iq_max_samples", cfg.iq_max_samples)
        if cfg.iq_channel_count != 1:
            raise ValueError(f"iq_channel_count must be 1 (multi-channel IQ files are not supported), got {cfg.iq_channel_count!r}")
        if cfg.true_label_mod is not None and cfg.true_label_mod not in RML2016_10A_MODULATIONS:
            raise ValueError(f"true_label_mod {cfg.true_label_mod!r} is not one of {RML2016_10A_MODULATIONS}")
    if cfg.iq_source == "radioml" and cfg.num_bursts <= 1:
        # Single-burst radioml mode only -- num_bursts>1 uses
        # dataset_mod_list/dataset_snr_list/sample_index_list instead
        # (checked below), and leaves these singular fields unused.
        missing = [n for n, v in (("dataset_path", cfg.dataset_path), ("dataset_mod", cfg.dataset_mod),
                                   ("dataset_snr", cfg.dataset_snr)) if v is None]
        if missing:
            raise ValueError(
                f"--iq-source radioml requires {missing} to all be set (none may be omitted)"
            )
        require_nonneg_int("sample_index", cfg.sample_index)
    elif cfg.iq_source == "radioml" and cfg.dataset_path is None:
        # Multi-burst mode still needs dataset_path (both branches load
        # from the same dataset file); dataset_mod/dataset_snr/sample_index
        # (singular) are not required here.
        raise ValueError("--iq-source radioml requires dataset_path to be set (none may be omitted)")
    require_positive_finite_float("embed_snr_margin", cfg.embed_snr_margin)
    # None is valid here -- it means "resolve the source-aware default",
    # done downstream in src/utils/pipeline.py (resolve_alignment_policy/
    # resolve_awn_preprocess below), which always produces a valid choice;
    # an explicitly-set value is still validated immediately.
    if cfg.alignment_policy is not None and cfg.alignment_policy not in ("naive", "max-energy"):
        raise ValueError(f"alignment_policy must be 'naive' or 'max-energy', got {cfg.alignment_policy!r}")
    require_positive_int("segment_hop", cfg.segment_hop)
    if cfg.awn_preprocess is not None and cfg.awn_preprocess not in ("legacy-unit-power", "radioml-native"):
        raise ValueError(
            f"awn_preprocess must be 'legacy-unit-power' or 'radioml-native', got {cfg.awn_preprocess!r}"
        )
    require_positive_int("num_bursts", cfg.num_bursts)
    require_nonneg_int("min_burst_gap", cfg.min_burst_gap)
    if cfg.max_burst_gap < cfg.min_burst_gap:
        raise ValueError(f"max_burst_gap ({cfg.max_burst_gap}) must be >= min_burst_gap ({cfg.min_burst_gap})")
    if cfg.num_bursts > 1:
        if cfg.iq_source != "radioml":
            raise ValueError("num_bursts > 1 requires --iq-source radioml")
        missing = [n for n, v in (("dataset_mod_list", cfg.dataset_mod_list),
                                   ("dataset_snr_list", cfg.dataset_snr_list),
                                   ("sample_index_list", cfg.sample_index_list)) if v is None]
        if missing:
            raise ValueError(f"num_bursts > 1 requires {missing} to all be set (none may be omitted)")
        for name, lst in (("dataset_mod_list", cfg.dataset_mod_list),
                           ("dataset_snr_list", cfg.dataset_snr_list),
                           ("sample_index_list", cfg.sample_index_list)):
            if len(lst) != cfg.num_bursts:
                raise ValueError(f"{name} has {len(lst)} entries, but num_bursts={cfg.num_bursts}")
        for idx in cfg.sample_index_list:
            require_nonneg_int("sample_index_list entry", idx)
        if cfg.burst_gap_list is not None:
            if len(cfg.burst_gap_list) != cfg.num_bursts:
                raise ValueError(f"burst_gap_list has {len(cfg.burst_gap_list)} entries, but num_bursts={cfg.num_bursts}")
            for gap in cfg.burst_gap_list:
                require_nonneg_int("burst_gap_list entry", gap)
        if cfg.burst_power_scale_list is not None:
            if len(cfg.burst_power_scale_list) != cfg.num_bursts:
                raise ValueError(
                    f"burst_power_scale_list has {len(cfg.burst_power_scale_list)} entries, "
                    f"but num_bursts={cfg.num_bursts}"
                )
            for scale in cfg.burst_power_scale_list:
                require_positive_finite_float("burst_power_scale_list entry", scale)

    # --- new this round: dataset / mod_filter / snr_filter / samples_per_cell /
    # batch_size / burst_insert_position / stream_length>=burst_len / strict
    # topk range. All are additive/None-safe -- verified against every
    # existing direct-API formal caller (run_phase1_sensing_baseline.py, the
    # only formal script that calls run_dry_run_experiment/
    # validate_experiment_config directly) before being added here; none of
    # these reject Phase 1's actual fixed params (dataset="RML2016.10a" by
    # construction, topk=50, mod_filter/snr_filter/samples_per_cell/
    # experiment_name all left at their None/default). min_region_len's
    # STRICT (>0) check is deliberately NOT here -- see
    # require_valid_min_region_len_strict()'s own docstring for why.
    if cfg.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"dataset must be one of {SUPPORTED_DATASETS}, got {cfg.dataset!r}")
    if cfg.mod_filter is not None:
        unknown = [m for m in cfg.mod_filter if m not in RML2016_10A_MODULATIONS]
        if unknown:
            raise ValueError(f"mod_filter contains unknown modulation(s) {unknown}; valid: {RML2016_10A_MODULATIONS}")
        if cfg.iq_source == "radioml" and cfg.dataset_mod is not None and cfg.dataset_mod not in cfg.mod_filter:
            raise ValueError(f"dataset_mod={cfg.dataset_mod!r} is not in mod_filter={cfg.mod_filter}")
    if cfg.snr_filter is not None:
        unknown = [s for s in cfg.snr_filter if s not in RML2016_10A_VALID_SNRS]
        if unknown:
            raise ValueError(f"snr_filter contains SNR(s) not present in RML2016.10a {unknown}; valid: {RML2016_10A_VALID_SNRS}")
        if cfg.iq_source == "radioml" and cfg.dataset_snr is not None and cfg.dataset_snr not in cfg.snr_filter:
            raise ValueError(f"dataset_snr={cfg.dataset_snr!r} is not in snr_filter={cfg.snr_filter}")
    if cfg.samples_per_cell is not None:
        require_positive_int("samples_per_cell", cfg.samples_per_cell)
        if cfg.iq_source == "radioml" and cfg.num_bursts <= 1 and cfg.sample_index >= cfg.samples_per_cell:
            raise ValueError(f"sample_index={cfg.sample_index} must be < samples_per_cell={cfg.samples_per_cell}")
    require_positive_int("batch_size", cfg.batch_size)
    if cfg.burst_insert_position not in ("random", "center", "explicit"):
        raise ValueError(f"burst_insert_position must be 'random', 'center', or 'explicit', got {cfg.burst_insert_position!r}")
    if cfg.burst_insert_position == "explicit":
        if cfg.burst_insert_position_index is None:
            raise ValueError("burst_insert_position='explicit' requires burst_insert_position_index to be set")
        require_nonneg_int("burst_insert_position_index", cfg.burst_insert_position_index)
        if cfg.burst_insert_position_index + cfg.window_size > cfg.n_samples:
            raise ValueError(
                f"burst_insert_position_index={cfg.burst_insert_position_index} + burst length "
                f"({cfg.window_size}) exceeds n_samples={cfg.n_samples} -- burst would not fit in the stream"
            )
    if cfg.n_samples < cfg.window_size:
        raise ValueError(f"n_samples (stream_length) = {cfg.n_samples} must be >= burst length ({cfg.window_size})")
    # Strict topk range -- SAFE to include in the shared validator (unlike
    # min_region_len above): Phase 1, the only formal round whose direct-API
    # call passes through this function, uses topk=50, well within [1,128];
    # every other formal round bypasses this validator entirely (calls
    # TopKAdapter.apply()/fft_topk_denoise directly, whose own bypass/clamp
    # semantics are untouched -- see require_valid_topk_strict's docstring).
    require_valid_topk_strict("topk", cfg.topk)
    # torch_num_threads: None is valid (means "leave default untouched");
    # an explicitly-set value must be a positive int -- 0 or negative is
    # never meaningful for torch.set_num_threads().
    if cfg.torch_num_threads is not None:
        require_positive_int("torch_num_threads", cfg.torch_num_threads)


def resolve_sensing_window_size(window_size: int, sensing_window_size: Optional[int]) -> int:
    """--sensing-window-size controls only energy_detect's smoothing window;
    --window-size (legacy name) continues to control segment_regions'/
    to_awn_input's seg_len (segment length == AWN input temporal length,
    UNCHANGED). When --sensing-window-size is unset (None), the effective
    sensing window falls back to window_size -- this is the single point
    where that fallback happens, called from
    src/utils/pipeline.py:run_dry_run_experiment so it applies uniformly
    regardless of whether the caller went through argparse or built
    ExperimentConfig directly."""
    return window_size if sensing_window_size is None else sensing_window_size


def resolve_alignment_policy(iq_source: str, alignment_policy: Optional[str]) -> str:
    """
    Source-aware default (docs/parameter_validation.md section 20). When
    --alignment-policy is left unset (None), radioml mode resolves to
    "max-energy" (section 18.4/19.4 validated this recovers oracle-path
    AMC accuracy on real RadioML samples) and synthetic mode resolves to
    "naive" (unchanged -- section 18's degradation diagnosis is specific to
    real embedded RadioML bursts; there's no equivalent finding, or need,
    for the synthetic generator's own cosmetic bursts). An explicitly
    passed alignment_policy is returned completely unchanged, regardless of
    iq_source -- this function only ever fills in a None.
    """
    if alignment_policy is not None:
        return alignment_policy
    resolved = "max-energy" if iq_source == "radioml" else "naive"
    print(f"[config] --alignment-policy unset; source-aware default for iq_source={iq_source!r} -> {resolved!r}")
    return resolved


def resolve_awn_preprocess(iq_source: str, awn_preprocess: Optional[str]) -> str:
    """
    Source-aware default (docs/parameter_validation.md section 20). When
    --awn-preprocess is left unset (None), radioml mode resolves to
    "radioml-native" (section 19.1's traced adversarial-rf evidence +
    19.4's validated accuracy recovery) and synthetic mode resolves to
    "legacy-unit-power" (unchanged -- the synthetic generator's amplitude
    convention was never compared against a real AWN training distribution,
    so there is no equivalent evidence to justify switching it). An
    explicitly passed awn_preprocess is returned completely unchanged,
    regardless of iq_source -- this function only ever fills in a None.
    """
    if awn_preprocess is not None:
        return awn_preprocess
    resolved = "radioml-native" if iq_source == "radioml" else "legacy-unit-power"
    print(f"[config] --awn-preprocess unset; source-aware default for iq_source={iq_source!r} -> {resolved!r}")
    return resolved


# ---------------------------------------------------------------------------
# argparse type= factories built on the same validators above, so a CLI
# parse-time error and a direct-API ValueError use identical wording. Reused
# by both build_arg_parser() below and experiments/run_batch.py's own parser.
# ---------------------------------------------------------------------------

def arg_positive_finite_float(name: str):
    def _parse(raw: str) -> float:
        try:
            value = float(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be a positive finite number, got {raw!r}")
        try:
            return require_positive_finite_float(name, value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))
    return _parse


def arg_finite_float(name: str):
    def _parse(raw: str) -> float:
        try:
            value = float(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be finite, got {raw!r}")
        try:
            return require_finite_float(name, value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))
    return _parse


def arg_nonneg_finite_float(name: str):
    def _parse(raw: str) -> float:
        try:
            value = float(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be a finite non-negative number, got {raw!r}")
        try:
            return require_nonneg_finite_float(name, value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))
    return _parse


def arg_positive_int(name: str):
    def _parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be a positive integer, got {raw!r}")
        try:
            return require_positive_int(name, value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))
    return _parse


def arg_nonneg_int(name: str):
    def _parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be a non-negative integer, got {raw!r}")
        try:
            return require_nonneg_int(name, value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))
    return _parse


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Argparse shared by experiments/run_full_experiment.py and run_batch.py (single-run flags)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--snr", type=arg_finite_float("snr_db"), default=10.0, help="Synthetic burst SNR in dB")
    parser.add_argument("--mod", type=str, default="BPSK", help="Modulation label tag (cosmetic only in this phase)")
    parser.add_argument("--attack", type=str, default="none", help="Attack name: none, fgsm, pgd, or cw")
    parser.add_argument("--attack-eps", type=arg_nonneg_finite_float("attack_eps"), default=0.03, help="Attack epsilon (Linf budget for fgsm/pgd)")
    parser.add_argument("--topk", type=int, default=50, help="Top-K FFT bins kept by the defense placeholder")
    parser.add_argument("--threshold-factor", type=arg_positive_finite_float("threshold_factor"), default=5.0, help="Energy threshold = median power * this factor")
    parser.add_argument("--window-size", type=arg_positive_int("window_size"), default=128,
                        help="Legacy name -- controls segment length AND AWN input temporal length "
                             "(segment_regions'/to_awn_input's seg_len). Real AWN checkpoint currently "
                             "expects 128 (not enforced here). Does NOT control energy-detection smoothing "
                             "window unless --sensing-window-size is left unset.")
    parser.add_argument("--sensing-window-size", type=arg_positive_int("sensing_window_size"), default=None,
                        help="Energy-detection smoothing window (energy_detect's window= argument), "
                             "independent of segment length / AWN input length. Defaults to --window-size "
                             "when unset, reproducing prior (coupled) behavior exactly.")
    parser.add_argument("--min-region-len", type=arg_nonneg_int("min_region_len"), default=None, help="Minimum occupied region length to keep (default: --window-size); 0 is allowed")
    parser.add_argument("--merge-gap", type=int, default=0, help="Merge occupied regions separated by <= this many samples")
    parser.add_argument("--burst-len", type=arg_positive_int("burst_len"), default=600, help="Synthetic burst length in samples")
    parser.add_argument("--output-dir", type=str, default="results/run", help="Directory for summary.csv and sensing_plot.png")
    parser.add_argument("--dry-run", action="store_true", help="Run the placeholder pipeline (required in this phase)")
    parser.add_argument("--use-real-topk", action="store_true",
                        help="Route the Top-K defense through TopKAdapter (real fft_topk_denoise if torch is "
                             "available, else falls back to the numpy dummy with notes in summary.csv)")
    parser.add_argument("--use-real-awn", action="store_true",
                        help="Route AWN inference through AWNModelAdapter (real AWN model + checkpoint if "
                             "torch is available, else falls back to the numpy dummy with notes in summary.csv)")
    parser.add_argument("--checkpoint", type=str, default="external/adversarial-rf/2016.10a_AWN.pkl",
                        help="Path to the AWN checkpoint (.pkl) used when --use-real-awn is set")
    parser.add_argument("--device", type=str, default="cpu", help="torch device for real AWN inference (cpu or cuda)")
    parser.add_argument("--use-real-attack", action="store_true",
                        help="Route the attack through AttackAdapter (real torchattacks-based attack if torch, "
                             "torchattacks, and a real AWN model are all available, else falls back to the "
                             "numpy dummy with notes in summary.csv)")
    parser.add_argument("--attack-temperature", type=arg_positive_finite_float("attack_temperature"), default=1.0,
                        help="Positive temperature T dividing AWN logits inside the attack's internal loss "
                             "only (attack_logits = logits / T); clean/attacked/defended inference elsewhere "
                             "always use raw logits. T=1.0 reproduces prior behavior (must be > 0).")
    parser.add_argument("--attack-diagnostics", action="store_true",
                        help="Run an extra diagnostic-only autograd.grad pass per real attack call to report "
                             "gradient nonzero-count/maxabs in summary.csv. Adds runtime cost per segment; "
                             "leave off for large batches.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Global reproducibility seed: seeds random/numpy/torch(+cuda if available) once "
                             "at the start of the run, and is threaded through to synthetic-IQ generation and "
                             "every dummy/real attack call. Default 0 reproduces prior (hardcoded) behavior.")
    parser.add_argument("--cw-c", type=arg_positive_finite_float("cw_c"), default=1.0,
                        help="CW-ONLY. torchattacks.CW's c (misclassification-loss weight). Ignored entirely "
                             "by fgsm/pgd. NOT the same knob as --attack-eps, which CW does not use at all.")
    parser.add_argument("--cw-steps", type=arg_positive_int("cw_steps"), default=20,
                        help="CW-ONLY. torchattacks.CW's optimization step count. Ignored entirely by fgsm/pgd.")
    parser.add_argument("--cw-lr", type=arg_positive_finite_float("cw_lr"), default=0.01,
                        help="CW-ONLY. torchattacks.CW's Adam learning rate. Ignored entirely by fgsm/pgd. "
                             "NOT the same knob as --attack-eps, which CW does not use at all.")
    parser.add_argument("--iq-source", "--input-source", dest="iq_source", type=str,
                        choices=["synthetic", "radioml", "cfile"], default="synthetic",
                        help="'synthetic' (default): generate_synthetic_iq, --mod/--snr control it as before. "
                             "'radioml': load a real RML2016.10a sample (--dataset-path/--dataset-mod/"
                             "--dataset-snr/--sample-index, all required) and embed it in a synthetic noise "
                             "stream instead -- --mod/--snr are ignored in this mode, not reinterpreted as "
                             "the RadioML ground truth. 'cfile': load a real IQ file via "
                             "src/io/iq_file_source.py (--input-path/--iq-format required; see those flags) -- "
                             "no synthetic stream, no RadioML sample, no ground truth. --input-source is an "
                             "alias for this same flag (same underlying iq_source field).")
    parser.add_argument("--input-path", dest="input_path", type=str, default=None,
                        help="CFILE-ONLY, REQUIRED when --iq-source cfile. Path to the raw IQ file.")
    parser.add_argument("--iq-format", type=str, choices=["complex64", "interleaved_float32", "interleaved_int16"],
                        default="complex64", help="CFILE-ONLY. On-disk format -- see src/io/iq_file_source.py.")
    parser.add_argument("--iq-endianness", type=str, choices=["little", "big", "native"], default="native",
                        help="CFILE-ONLY. Byte order of the on-disk samples.")
    parser.add_argument("--iq-scale", type=float, default=None,
                        help="CFILE-ONLY. interleaved_int16 dequantization scale (complex_value = raw_int16 * scale). "
                             "Default (unset): 1/32768, always recorded in the output provenance either way.")
    parser.add_argument("--iq-sample-rate", type=float, default=None,
                        help="CFILE-ONLY. Sample rate (Hz) of the capture, recorded as provenance metadata only "
                             "(not used by any sensing/AWN computation).")
    parser.add_argument("--iq-offset-samples", type=arg_nonneg_int("iq_offset_samples"), default=0,
                        help="CFILE-ONLY. Skip this many samples from the start of the file before loading.")
    parser.add_argument("--iq-max-samples", type=arg_positive_int("iq_max_samples"), default=None,
                        help="CFILE-ONLY. Load at most this many samples (after --iq-offset-samples). Default: all.")
    parser.add_argument("--iq-channel-count", type=int, default=1,
                        help="CFILE-ONLY. Must be 1 -- multi-channel IQ files are not supported by this loader.")
    parser.add_argument("--true-label-mod", type=str, default=None,
                        help="CFILE-ONLY, OPTIONAL. If the true modulation of a captured file happens to be "
                             "known, supply it here to enable accuracy/attack-success metrics; omitted (default) "
                             "leaves those metrics unavailable (None), never fabricated.")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="RADIOML-ONLY, REQUIRED when --iq-source radioml. Absolute path to "
                             "RML2016.10a_dict.pkl (not part of this repo or its submodule).")
    parser.add_argument("--dataset-mod", type=str, default=None,
                        help="RADIOML-ONLY, REQUIRED when --iq-source radioml. Real RadioML modulation label "
                             "to select from the dataset (e.g. QPSK, BPSK) -- distinct from --mod, which only "
                             "affects the synthetic generator and is unused in radioml mode.")
    parser.add_argument("--dataset-snr", type=int, default=None,
                        help="RADIOML-ONLY, REQUIRED when --iq-source radioml. Real RadioML SNR label (dB, "
                             "one of -20..18 in steps of 2) to select from the dataset -- distinct from --snr, "
                             "which only affects the synthetic generator and is unused in radioml mode.")
    parser.add_argument("--sample-index", type=arg_nonneg_int("sample_index"), default=0,
                        help="RADIOML-ONLY. Index within the selected (dataset-mod, dataset-snr) block of "
                             "1000 samples.")
    parser.add_argument("--embed-snr-margin", type=arg_positive_finite_float("embed_snr_margin"), default=20.0,
                        help="RADIOML-ONLY. How much the embedded RadioML burst's own power exceeds the "
                             "surrounding synthetic capture-noise floor (src/sensing/radioml_source.py). "
                             "Distinct from --dataset-snr, which is the sample's own baked-in label SNR.")
    parser.add_argument("--num-bursts", type=arg_positive_int("num_bursts"), default=1,
                        help="RADIOML-ONLY. 1 (default): exact same single-burst code path as before, using "
                             "--dataset-mod/--dataset-snr/--sample-index. >1: multi-burst mode, requires "
                             "--dataset-mod-list/--dataset-snr-list/--sample-index-list (comma-separated, "
                             "each with exactly --num-bursts entries) instead.")
    parser.add_argument("--dataset-mod-list", type=str, default=None,
                        help="MULTI-BURST-ONLY, REQUIRED when --num-bursts > 1. Comma-separated RadioML "
                             "modulation labels, one per burst, e.g. QPSK,BPSK,QPSK.")
    parser.add_argument("--dataset-snr-list", type=str, default=None,
                        help="MULTI-BURST-ONLY, REQUIRED when --num-bursts > 1. Comma-separated RadioML SNR "
                             "labels (dB), one per burst, e.g. 18,0,18.")
    parser.add_argument("--sample-index-list", type=str, default=None,
                        help="MULTI-BURST-ONLY, REQUIRED when --num-bursts > 1. Comma-separated sample "
                             "indices, one per burst, e.g. 0,1,2.")
    parser.add_argument("--min-burst-gap", type=arg_nonneg_int("min_burst_gap"), default=50,
                        help="MULTI-BURST-ONLY. Minimum gap (samples) drawn before each burst (including a "
                             "leading gap before the first). Set equal to --max-burst-gap for an exact, "
                             "deterministic gap (used by the merge-gap main-pipeline test cases).")
    parser.add_argument("--max-burst-gap", type=arg_nonneg_int("max_burst_gap"), default=50,
                        help="MULTI-BURST-ONLY. Maximum gap (samples) drawn before each burst. Must be >= "
                             "--min-burst-gap.")
    parser.add_argument("--burst-gap-list", type=str, default=None,
                        help="MULTI-BURST-ONLY. Comma-separated EXACT gap (samples) before each burst "
                             "(length must equal --num-bursts), overriding --min-burst-gap/--max-burst-gap's "
                             "random sampling entirely. Needed when different bursts need different gaps in "
                             "the same run (e.g. one pair close enough to merge, another far enough apart to "
                             "stay separate).")
    parser.add_argument("--burst-power-scale-list", type=str, default=None,
                        help="MULTI-BURST-ONLY. Comma-separated per-burst amplitude multiplier (length must "
                             "equal --num-bursts, each > 0), applied before the shared noise floor is "
                             "computed. Used to construct a genuinely low-energy burst on demand for "
                             "detection-boundary tests -- real RadioML samples alone have only modest "
                             "power differences across mod/snr.")
    parser.add_argument("--alignment-policy", type=str, choices=["naive", "max-energy"], default=None,
                        help="Segment-alignment policy (src/sensing/segmentation.py:select_aligned_segments). "
                             "Default (unset): SOURCE-AWARE -- 'max-energy' for --iq-source radioml, 'naive' for "
                             "'synthetic' (docs/parameter_validation.md section 20). 'naive': identical to every "
                             "pre-round-9 behavior -- fixed non-overlapping windows from each detected region's "
                             "own start. 'max-energy': one highest-mean-power seg_len window per region, chosen "
                             "by sliding search (never references true burst position). See section 18.")
    parser.add_argument("--segment-hop", type=arg_positive_int("segment_hop"), default=1,
                        help="Sliding-window step (samples) for max-energy's candidate search (and reported, "
                             "informationally, as candidate_count under naive too). Default 1 (every offset).")
    parser.add_argument("--awn-preprocess", type=str, choices=["legacy-unit-power", "radioml-native"],
                        default=None,
                        help="AWN-input-boundary preprocessing (src/sensing/normalize.py:apply_awn_preprocess). "
                             "Default (unset): SOURCE-AWARE -- 'radioml-native' for --iq-source radioml, "
                             "'legacy-unit-power' for 'synthetic' (docs/parameter_validation.md section 20). "
                             "'legacy-unit-power': normalize_segments()'s unit-average-power rescale. "
                             "'radioml-native': no rescaling at all, matching traced evidence that "
                             "external/adversarial-rf never normalizes before AWN.forward(). See section 19.")

    # Extended attack parameter surface (this round) -- see ExperimentConfig's
    # attack_* fields and build_attack_params() above for how these feed
    # AttackAdapter.apply(attack_params=...). Every one defaults to None
    # (unset): omitting a flag never changes fgsm/pgd/cw's existing behavior,
    # and for every other attack lets torchattacks' own installed default
    # apply. docs/ATTACK_NAME_MAPPING.md documents which flags apply to
    # which attack name.
    parser.add_argument("--attack-alpha", type=float, default=None, help="pgd/bim/mifgsm/difgsm/vmifgsm/vnifgsm/rfgsm/tpgd step size")
    parser.add_argument("--attack-steps", type=arg_positive_int("attack_steps"), default=None, help="iterative attacks' step count (most families)")
    attack_random_start_group = parser.add_mutually_exclusive_group()
    attack_random_start_group.add_argument("--attack-random-start", dest="attack_random_start", action="store_true", default=None, help="pgd/difgsm: randomize the starting point before the first step")
    attack_random_start_group.add_argument("--attack-no-random-start", dest="attack_random_start", action="store_false", help="pgd/difgsm: disable random start")
    parser.add_argument("--attack-decay", type=float, default=None, help="mifgsm/difgsm/vmifgsm/vnifgsm momentum decay")
    parser.add_argument("--attack-resize-rate", type=float, default=None, help="difgsm input-diversity resize rate")
    parser.add_argument("--attack-diversity-prob", type=float, default=None, help="difgsm input-diversity probability")
    parser.add_argument("--attack-momentum-n", type=arg_positive_int("attack_momentum_n"), default=None, help="vmifgsm/vnifgsm neighborhood sample count (torchattacks' N)")
    parser.add_argument("--attack-beta", type=float, default=None, help="fab step-size beta OR ead L1-regularization beta (mutually exclusive per attack, see docs/ATTACK_NAME_MAPPING.md)")
    parser.add_argument("--attack-overshoot", type=arg_positive_finite_float("attack_overshoot"), default=None, help="deepfool overshoot")
    parser.add_argument("--attack-kappa", type=float, default=None, help="cw/ead confidence margin kappa")
    parser.add_argument("--attack-lr", type=arg_positive_finite_float("attack_lr"), default=None, help="ead Adam learning rate (distinct from --cw-lr)")
    parser.add_argument("--attack-norm", type=str, choices=["Linf", "L2", "L1"], default=None, help="fab/square/apgd/apgdt/autoattack distance norm")
    parser.add_argument("--attack-n-restarts", type=arg_positive_int("attack_n_restarts"), default=None, help="fab/square/apgd/apgdt restart count")
    parser.add_argument("--attack-loss", type=str, choices=["ce", "dlr"], default=None, help="apgd loss function")
    parser.add_argument("--attack-eot-iter", type=arg_positive_int("attack_eot_iter"), default=None, help="apgd/apgdt EOT iteration count")
    parser.add_argument("--attack-rho", type=arg_positive_finite_float("attack_rho"), default=None, help="apgd/apgdt step-size-reduction parameter")
    parser.add_argument("--attack-alpha-max", type=arg_positive_finite_float("attack_alpha_max"), default=None, help="fab max step size")
    parser.add_argument("--attack-eta", type=arg_positive_finite_float("attack_eta"), default=None, help="fab step-size growth factor")
    attack_multi_targeted_group = parser.add_mutually_exclusive_group()
    attack_multi_targeted_group.add_argument("--attack-multi-targeted", dest="attack_multi_targeted", action="store_true", default=None, help="fab: attack every other class as a target and keep the best")
    attack_multi_targeted_group.add_argument("--attack-no-multi-targeted", dest="attack_multi_targeted", action="store_false", help="fab: disable multi-targeted mode")
    parser.add_argument("--attack-n-classes", type=arg_positive_int("attack_n_classes"), default=None, help="fab/apgdt/autoattack class count (default: AWN's real 11 if unset, NOT torchattacks' CIFAR-10-oriented default of 10)")
    parser.add_argument("--attack-internal-seed", type=int, default=None, help="fab/square/apgd/apgdt/autoattack's own internal RNG seed (distinct from --seed)")
    parser.add_argument("--attack-n-queries", type=arg_positive_int("attack_n_queries"), default=None, help="square query budget")
    parser.add_argument("--attack-p-init", type=arg_positive_finite_float("attack_p_init"), default=None, help="square initial perturbation fraction")
    attack_resc_group = parser.add_mutually_exclusive_group()
    attack_resc_group.add_argument("--attack-resc-schedule", dest="attack_resc_schedule", action="store_true", default=None, help="square: rescale the query budget schedule")
    attack_resc_group.add_argument("--attack-no-resc-schedule", dest="attack_resc_schedule", action="store_false", help="square: disable query budget rescaling")
    parser.add_argument("--attack-version", type=str, choices=["standard", "plus", "rand"], default=None, help="autoattack ensemble version")
    parser.add_argument("--attack-binary-search-steps", type=arg_positive_int("attack_binary_search_steps"), default=None, help="ead binary search step count")
    parser.add_argument("--attack-max-iterations", type=arg_positive_int("attack_max_iterations"), default=None, help="ead max optimization iterations")
    parser.add_argument("--attack-initial-const", type=arg_positive_finite_float("attack_initial_const"), default=None, help="ead initial trade-off constant")
    attack_abort_group = parser.add_mutually_exclusive_group()
    attack_abort_group.add_argument("--attack-abort-early", dest="attack_abort_early", action="store_true", default=None, help="ead: stop a binary-search step early once successful")
    attack_abort_group.add_argument("--attack-no-abort-early", dest="attack_abort_early", action="store_false", help="ead: always run the full iteration budget")
    parser.add_argument("--attack-ead-variant", type=str, choices=["eadl1", "eaden"], default=None, help="which torchattacks class 'ead' aliases (default eadl1)")

    # Dataset / sample-selection / runtime-management surface (this round).
    parser.add_argument("--dataset", type=str, choices=list(SUPPORTED_DATASETS), default="RML2016.10a",
                         help="Only RML2016.10a is supported end-to-end; any other value is rejected before any work starts.")
    parser.add_argument("--mod-filter", type=str, default=None,
                         help="Comma-separated whitelist of modulations dataset_mod must belong to (e.g. QPSK,BPSK,QAM16). "
                              "Not a batch-iteration driver -- a guard on the single selected --dataset-mod.")
    parser.add_argument("--snr-filter", type=str, default=None,
                         help="Comma-separated whitelist of SNRs (dB) dataset_snr must belong to. Same guard semantics as --mod-filter.")
    parser.add_argument("--samples-per-cell", type=arg_positive_int("samples_per_cell"), default=None,
                         help="Upper bound on --sample-index for the selected (dataset-mod, dataset-snr) cell "
                              "(sample_index must be < this). Distinct from --stream-length (below), which is an "
                              "unrelated pre-existing concept (long synthetic-stream sample count).")
    parser.add_argument("--stream-length", dest="n_samples", type=arg_positive_int("n_samples"), default=8192,
                         help="Long synthetic-noise-stream length in samples (the pre-existing n_samples field, "
                              "now CLI-settable) -- must be >= the burst/segment length (--window-size, 128 by default).")
    parser.add_argument("--burst-insert-position", type=str, choices=["random", "center", "explicit"], default="random",
                         help="RadioML single-burst mode only. 'random' (default) reproduces the existing seeded-random "
                              "placement exactly; 'center' places the burst at the stream's geometric center; 'explicit' "
                              "requires --burst-insert-position-index.")
    parser.add_argument("--burst-insert-position-index", type=arg_nonneg_int("burst_insert_position_index"), default=None,
                         help="REQUIRED when --burst-insert-position explicit. Exact burst start sample index; "
                              "index + burst length must not exceed --stream-length.")
    parser.add_argument("--batch-size", type=arg_positive_int("batch_size"), default=1,
                         help="How many segments AWNModelAdapter.infer() processes per real forward-pass call when "
                              "more than one segment is available; chunks larger segment arrays into sub-batches of "
                              "this size instead of one giant or one-at-a-time call.")
    parser.add_argument("--torch-threads", dest="torch_num_threads", type=arg_positive_int("torch_num_threads"), default=None,
                         help="If set, calls torch.set_num_threads(value) once near the top of the run, before any "
                              "AWN/attack backend is constructed. Default: None, leaves torch's own/environment "
                              "default thread count untouched (prior behavior, unchanged). Must be a positive "
                              "integer; 0 or negative is rejected.")
    parser.add_argument("--experiment-name", type=str, default=None,
                         help="Optional human-readable tag folded into the manifest/output (sanitized -- path "
                              "separators and other illegal filename characters are stripped).")
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument("--overwrite", dest="overwrite", action="store_true", default=False,
                                  help="Allow writing into an --output-dir that already contains a summary.csv "
                                       "(default: refuse). Only ever touches the exact output_dir given.")
    overwrite_group.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    return parser


def _parse_comma_list(raw: Optional[str], cast, name: str) -> Optional[List]:
    """Shared comma-list parser for the multi-burst CLI flags above --
    distinct from experiments/run_batch.py's own _parse_list, which drives
    that script's (snr, mod, attack, topk) BATCH grid, a different concept
    from a single run's list of per-burst specs. Returns None if raw is
    None (flag omitted); raises argparse.ArgumentTypeError with a specific
    element's value on a cast failure, not a generic parse error."""
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    try:
        return [cast(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name}: could not parse {raw!r} ({exc})")


def args_to_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        snr=args.snr,
        mod=args.mod,
        attack=args.attack,
        topk=args.topk,
        threshold_factor=args.threshold_factor,
        window_size=args.window_size,
        sensing_window_size=args.sensing_window_size,
        min_region_len=(
            args.window_size if args.min_region_len is None
            else require_valid_min_region_len_strict("min_region_len", args.min_region_len)
        ),
        merge_gap=args.merge_gap,
        burst_len=args.burst_len,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        use_real_topk=args.use_real_topk,
        use_real_awn=args.use_real_awn,
        checkpoint=args.checkpoint,
        device=args.device,
        attack_eps=args.attack_eps,
        use_real_attack=args.use_real_attack,
        attack_temperature=args.attack_temperature,
        attack_diagnostics=args.attack_diagnostics,
        seed=args.seed,
        cw_c=args.cw_c,
        cw_steps=args.cw_steps,
        cw_lr=args.cw_lr,
        iq_source=args.iq_source,
        dataset_path=args.dataset_path,
        dataset_mod=args.dataset_mod,
        dataset_snr=args.dataset_snr,
        sample_index=args.sample_index,
        embed_snr_margin=args.embed_snr_margin,
        num_bursts=args.num_bursts,
        dataset_mod_list=_parse_comma_list(args.dataset_mod_list, str, "dataset_mod_list"),
        dataset_snr_list=_parse_comma_list(args.dataset_snr_list, int, "dataset_snr_list"),
        sample_index_list=_parse_comma_list(args.sample_index_list, int, "sample_index_list"),
        min_burst_gap=args.min_burst_gap,
        max_burst_gap=args.max_burst_gap,
        burst_gap_list=_parse_comma_list(args.burst_gap_list, int, "burst_gap_list"),
        burst_power_scale_list=_parse_comma_list(args.burst_power_scale_list, float, "burst_power_scale_list"),
        alignment_policy=args.alignment_policy,
        segment_hop=args.segment_hop,
        awn_preprocess=args.awn_preprocess,
        attack_alpha=args.attack_alpha,
        attack_steps=args.attack_steps,
        attack_random_start=args.attack_random_start,
        attack_decay=args.attack_decay,
        attack_resize_rate=args.attack_resize_rate,
        attack_diversity_prob=args.attack_diversity_prob,
        attack_momentum_n=args.attack_momentum_n,
        attack_beta=args.attack_beta,
        attack_overshoot=args.attack_overshoot,
        attack_kappa=args.attack_kappa,
        attack_lr=args.attack_lr,
        attack_norm=args.attack_norm,
        attack_n_restarts=args.attack_n_restarts,
        attack_loss=args.attack_loss,
        attack_eot_iter=args.attack_eot_iter,
        attack_rho=args.attack_rho,
        attack_alpha_max=args.attack_alpha_max,
        attack_eta=args.attack_eta,
        attack_multi_targeted=args.attack_multi_targeted,
        attack_n_classes=args.attack_n_classes,
        attack_internal_seed=args.attack_internal_seed,
        attack_n_queries=args.attack_n_queries,
        attack_p_init=args.attack_p_init,
        attack_resc_schedule=args.attack_resc_schedule,
        attack_version=args.attack_version,
        attack_binary_search_steps=args.attack_binary_search_steps,
        attack_max_iterations=args.attack_max_iterations,
        attack_initial_const=args.attack_initial_const,
        attack_abort_early=args.attack_abort_early,
        attack_ead_variant=args.attack_ead_variant,
        dataset=args.dataset,
        mod_filter=_parse_comma_list(args.mod_filter, str, "mod_filter"),
        snr_filter=_parse_comma_list(args.snr_filter, int, "snr_filter"),
        samples_per_cell=args.samples_per_cell,
        n_samples=args.n_samples,
        burst_insert_position=args.burst_insert_position,
        burst_insert_position_index=args.burst_insert_position_index,
        batch_size=args.batch_size,
        torch_num_threads=args.torch_num_threads,
        experiment_name=sanitize_experiment_name(args.experiment_name),
        overwrite=args.overwrite,
        input_path=args.input_path,
        iq_format=args.iq_format,
        iq_endianness=args.iq_endianness,
        iq_scale=args.iq_scale,
        iq_sample_rate=args.iq_sample_rate,
        iq_offset_samples=args.iq_offset_samples,
        iq_max_samples=args.iq_max_samples,
        iq_channel_count=args.iq_channel_count,
        true_label_mod=args.true_label_mod,
    )


def sanitize_experiment_name(name: Optional[str]) -> Optional[str]:
    """Strips path separators and other characters illegal/dangerous in a
    filename component -- experiment_name is folded into manifests/output
    paths, never used blindly as a raw path fragment."""
    if name is None:
        return None
    import re
    cleaned = re.sub(r"[^A-Za-z0-9_.\-]", "_", name)
    if not cleaned:
        raise ValueError(f"experiment_name {name!r} sanitizes to an empty string")
    return cleaned
