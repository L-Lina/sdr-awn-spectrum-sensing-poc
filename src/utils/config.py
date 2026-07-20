"""Shared experiment configuration and argparse definition for the runners in experiments/."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional


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
    # docs/parameter_validation.md section 18). "naive" (default) reproduces
    # every prior round's segment_regions() behavior byte-for-byte -- fixed,
    # non-overlapping seg_len windows starting at each detected region's own
    # start, which can be 53-61 samples before the true burst start (energy_detect's
    # smoothing widens the region's leading edge), so a region with 100%
    # region-level captured_signal_ratio can still yield a segment that's only
    # ~52-63% true-burst signal. "max-energy" selects, per region, the single
    # seg_len sliding window (hop=segment_hop) with the highest mean power --
    # never references true_burst_start/true_burst_end.
    alignment_policy: str = "naive"
    # Sliding-window step (samples) used by max-energy's candidate search
    # (and reported, informationally, as candidate_count even under naive).
    # Default 1 (every possible offset) for correctness testing; a batch run
    # over many combos may want a larger hop to reduce candidate-search cost.
    segment_hop: int = 1
    # AWN-input-boundary preprocessing policy (src/sensing/normalize.py:
    # apply_awn_preprocess, docs/parameter_validation.md section 19).
    # "legacy-unit-power" (DEFAULT, UNCHANGED this round pending a separate
    # decision) -- current normalize_segments() behavior. "radioml-native" --
    # no rescaling at all, matching traced evidence that
    # external/adversarial-rf never normalizes between its dataset loader
    # and AWN.forward(). Applied ONLY at the AWN input boundary in
    # src/utils/pipeline.py -- never inside segmentation.py/energy_detection.py.
    awn_preprocess: str = "legacy-unit-power"


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
    if cfg.iq_source not in ("synthetic", "radioml"):
        raise ValueError(f"iq_source must be 'synthetic' or 'radioml', got {cfg.iq_source!r}")
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
    if cfg.alignment_policy not in ("naive", "max-energy"):
        raise ValueError(f"alignment_policy must be 'naive' or 'max-energy', got {cfg.alignment_policy!r}")
    require_positive_int("segment_hop", cfg.segment_hop)
    if cfg.awn_preprocess not in ("legacy-unit-power", "radioml-native"):
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
    parser.add_argument("--iq-source", type=str, choices=["synthetic", "radioml"], default="synthetic",
                        help="'synthetic' (default): generate_synthetic_iq, --mod/--snr control it as before. "
                             "'radioml': load a real RML2016.10a sample (--dataset-path/--dataset-mod/"
                             "--dataset-snr/--sample-index, all required) and embed it in a synthetic noise "
                             "stream instead -- --mod/--snr are ignored in this mode, not reinterpreted as "
                             "the RadioML ground truth.")
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
    parser.add_argument("--alignment-policy", type=str, choices=["naive", "max-energy"], default="naive",
                        help="Segment-alignment policy (src/sensing/segmentation.py:select_aligned_segments). "
                             "'naive' (default): identical to every prior round's segment_regions() behavior -- "
                             "fixed non-overlapping windows from each detected region's own start. "
                             "'max-energy': one highest-mean-power seg_len window per region, chosen by sliding "
                             "search (never references true burst position). See "
                             "docs/parameter_validation.md section 18.")
    parser.add_argument("--segment-hop", type=arg_positive_int("segment_hop"), default=1,
                        help="Sliding-window step (samples) for max-energy's candidate search (and reported, "
                             "informationally, as candidate_count under naive too). Default 1 (every offset).")
    parser.add_argument("--awn-preprocess", type=str, choices=["legacy-unit-power", "radioml-native"],
                        default="legacy-unit-power",
                        help="AWN-input-boundary preprocessing (src/sensing/normalize.py:apply_awn_preprocess). "
                             "'legacy-unit-power' (default, unchanged this round): current normalize_segments() "
                             "unit-average-power rescale. 'radioml-native': no rescaling at all, matching traced "
                             "evidence that external/adversarial-rf never normalizes before AWN.forward(). See "
                             "docs/parameter_validation.md section 19.")
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
            args.window_size if args.min_region_len is None else args.min_region_len
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
    )
