"""
RadioML (RML2016.10a) IQ source: loads a real labeled [2,128] sample from the
official RML2016.10a_dict.pkl and embeds it at a known (to us, not to the
sensing stage) position inside a longer synthetic noise stream, so it can be
run through the exact same energy_detect -> segment -> AWN pipeline as the
existing synthetic source.

Does not modify external/AWN or external/adversarial-rf; only reads the
dataset file (an external, absolute filesystem path, not part of this repo
or its submodule -- see docs/parameter_validation.md for the RadioML
inventory).

Class ordering: RML2016.10a's official 11-class label mapping used to train
the pinned AWN checkpoint. Verified this session against THREE independent
declarations in external/adversarial-rf (submodule, commit ced705e), not
recalled from memory:
  - data_loader/data_loader.py:13-14 (Load_Dataset, actually called from
    main.py:205 -- the real training entry point)
  - util/config.py:52 (Config class, independent duplicate)
  - 6 further plot_*.py analysis scripts, all consistent
No conflicting ordering was found anywhere in the submodule.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

RML2016_10A_CLASSES: Dict[str, int] = {
    "QAM16": 0, "QAM64": 1, "8PSK": 2, "WBFM": 3, "BPSK": 4,
    "CPFSK": 5, "AM-DSB": 6, "GFSK": 7, "PAM4": 8, "QPSK": 9, "AM-SSB": 10,
}


def load_radioml_dict(dataset_path: str) -> dict:
    """Loads the raw {(mod: str, snr: int): ndarray[1000,2,128]} dict.
    Uses encoding='latin1' (str keys) rather than external/adversarial-rf's
    own encoding='bytes' choice (data_loader.py:36) -- both decode the same
    underlying Python-2-pickled data, this just avoids bytes/str key
    juggling in this repo's own CLI (--dataset-mod QPSK, a plain string).
    Takes several seconds for the full ~640MB file; no caching is done here
    -- each call re-reads from disk."""
    path = Path(dataset_path)
    if not path.exists():
        raise ValueError(f"RadioML dataset file not found: {dataset_path}")
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_radioml_sample(dataset_path: str, mod: str, snr: int, sample_index: int) -> np.ndarray:
    """Returns the raw [2, 128] float32 array for one (mod, snr, sample_index)
    triple. Raises ValueError (not a silent fallback) for any unknown mod,
    unknown snr, or out-of-range sample_index -- lists the valid options in
    the error message rather than guessing."""
    data = load_radioml_dict(dataset_path)
    available_mods = sorted({k[0] for k in data.keys()})
    available_snrs = sorted({k[1] for k in data.keys()})

    if mod not in available_mods:
        raise ValueError(f"Unknown RadioML modulation {mod!r}; available: {available_mods}")
    if snr not in available_snrs:
        raise ValueError(f"Unknown RadioML SNR {snr!r}; available: {available_snrs}")

    block = data[(mod, snr)]
    if sample_index < 0 or sample_index >= block.shape[0]:
        raise ValueError(
            f"sample_index {sample_index} out of range for ({mod}, {snr}); "
            f"block has {block.shape[0]} samples (valid range: 0..{block.shape[0] - 1})"
        )
    sample = block[sample_index]
    if sample.shape != (2, 128):
        raise ValueError(f"Unexpected RadioML sample shape {sample.shape}, expected (2, 128)")
    return sample.astype(np.float32)


def radioml_sample_to_iq(sample_2x128: np.ndarray) -> np.ndarray:
    """[2, 128] float32 (I row, Q row) -> [128] complex64, matching this
    repo's IQ convention (src/sensing/iq_source.py)."""
    return (sample_2x128[0] + 1j * sample_2x128[1]).astype(np.complex64)


def embed_sample_in_noise(
    sample_2x128: np.ndarray,
    n_samples: int,
    embed_snr_margin: float,
    seed: int,
) -> Tuple[np.ndarray, dict]:
    """Embeds a real RadioML [2,128] sample into a longer complex64 noise
    stream of length n_samples, at a seeded-random (reproducible, but not
    looked up by the sensing stage) position.

    Background noise power is set to (burst_mean_power / embed_snr_margin)
    -- RadioML samples already have SNR-dependent power baked in at a scale
    that is roughly mod/snr-independent in absolute terms (observed ~7e-5 to
    ~3.5e-4 mean power across mod/snr combos this session), so a fixed
    background noise std would make detectability depend on which (mod,snr)
    was picked. Scaling relative to the actual loaded sample's own power
    keeps detectability consistent across any (mod, snr) choice --
    embed_snr_margin is this repo's own "how much does the burst stand out
    above the surrounding capture noise floor" knob, deliberately distinct
    from the RadioML sample's own internal (mod, snr)-label SNR.

    Returns (iq, meta) where meta includes true_start/true_end and the
    embedding parameters actually used.
    """
    burst_len = sample_2x128.shape[1]
    if burst_len >= n_samples:
        raise ValueError(f"RadioML sample length ({burst_len}) must be < n_samples ({n_samples})")

    burst_iq = radioml_sample_to_iq(sample_2x128)
    burst_power = float(np.mean(np.abs(burst_iq) ** 2))
    noise_power = burst_power / embed_snr_margin
    noise_std = float(np.sqrt(noise_power / 2.0))  # split across real/imag

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, n_samples) + 1j * rng.normal(0, noise_std, n_samples)
    iq = noise.astype(np.complex64)

    max_start = n_samples - burst_len
    true_start = int(rng.integers(0, max_start + 1))
    true_end = true_start + burst_len
    iq[true_start:true_end] += burst_iq

    meta = {
        "true_start": true_start,
        "true_end": true_end,
        "burst_len": burst_len,
        "burst_power": burst_power,
        "embed_noise_power": noise_power,
        "embed_noise_std": noise_std,
        "embed_snr_margin": embed_snr_margin,
        "n_samples": n_samples,
    }
    print(f"[radioml] embedded {burst_len}-sample RadioML burst at [{true_start}:{true_end}] "
          f"in {n_samples}-sample stream (burst_power={burst_power:.3e}, "
          f"embed_noise_power={noise_power:.3e}, margin={embed_snr_margin})")
    return iq, meta


def embed_multiple_samples_in_noise(
    samples: List[np.ndarray],
    n_samples: int,
    embed_snr_margin: float,
    seed: int,
    min_burst_gap: int,
    max_burst_gap: int,
    gap_list: Optional[List[int]] = None,
    power_scale_list: Optional[List[float]] = None,
) -> Tuple[np.ndarray, List[dict]]:
    """Embeds MULTIPLE real RadioML [2,128] samples into one longer complex64
    noise stream, back-to-back with a gap before each burst (including a
    "leading gap" before the first one).

    By default (gap_list=None), each gap is drawn independently, uniformly
    from [min_burst_gap, max_burst_gap], seeded by `seed`. Setting
    min_burst_gap == max_burst_gap makes every gap an EXACT, fully
    deterministic value this way -- sufficient for test cases that need
    the same fixed gap everywhere.

    gap_list (optional): an exact, per-burst list of gaps (length ==
    len(samples)), overriding the random min/max sampling entirely --
    needed when different gaps are required at different positions in the
    same run (e.g. docs/parameter_validation.md section 15's Case 3: burst
    0->1 gap small enough to merge, burst 1->2 gap large enough to stay
    separate, which a single shared [min,max] range cannot express).
    min_burst_gap/max_burst_gap are still validated even when gap_list is
    given (for consistent error messages), but their values are otherwise
    unused in that case.

    Bursts are placed strictly back-to-back, never overlapping, by
    construction -- there is no separate overlap check needed as long as
    every gap (whichever source) is >= 0.

    A single background noise level is used for the WHOLE stream (one real
    capture has one noise floor) -- its power is set relative to the MEAN
    power across all the loaded bursts (not each one individually), so
    mixing bursts of different (mod, snr) does not bias detectability
    towards whichever burst happens to be louder.

    power_scale_list (optional): per-burst amplitude multiplier (length ==
    len(samples), each > 0), applied BEFORE the shared noise floor is
    computed -- so scaling one burst down genuinely makes it a weaker
    signal relative to the (still shared, single-capture) noise floor,
    rather than just relabeling it. Real RadioML samples across (mod, snr)
    combinations have only modest inherent power differences (observed
    ~1.35x block-mean spread this session -- not enough to reliably produce
    an "undetected" burst on its own), so this is the deliberate, explicit
    way to construct a genuinely low-energy burst for a controlled test
    (docs/parameter_validation.md section 15's Case 4) rather than hunting
    for a naturally-quiet dataset sample. Defaults to 1.0 (unscaled) for
    every burst, i.e. no effect, when omitted.

    Returns (iq, per_burst_meta) where per_burst_meta is a list of one dict
    per input sample, each with burst_id/true_start/true_end/gap_before_burst
    (caller is expected to merge in dataset_mod/dataset_snr/sample_index/
    original_sample_sha256 per burst, since this function only sees raw
    arrays, not their dataset labels).
    """
    if not samples:
        raise ValueError("embed_multiple_samples_in_noise requires at least one sample")
    if min_burst_gap < 0:
        raise ValueError(f"min_burst_gap must be >= 0, got {min_burst_gap}")
    if max_burst_gap < min_burst_gap:
        raise ValueError(f"max_burst_gap ({max_burst_gap}) must be >= min_burst_gap ({min_burst_gap})")
    if gap_list is not None:
        if len(gap_list) != len(samples):
            raise ValueError(f"gap_list has {len(gap_list)} entries, but there are {len(samples)} samples")
        if any(g < 0 for g in gap_list):
            raise ValueError(f"gap_list entries must all be >= 0, got {gap_list}")
    if power_scale_list is not None:
        if len(power_scale_list) != len(samples):
            raise ValueError(f"power_scale_list has {len(power_scale_list)} entries, but there are {len(samples)} samples")
        if any(s <= 0 for s in power_scale_list):
            raise ValueError(f"power_scale_list entries must all be > 0, got {power_scale_list}")

    bursts_iq = [radioml_sample_to_iq(s) for s in samples]
    if power_scale_list is not None:
        bursts_iq = [b * scale for b, scale in zip(bursts_iq, power_scale_list)]
    burst_lens = [len(b) for b in bursts_iq]
    mean_burst_power = float(np.mean([np.mean(np.abs(b) ** 2) for b in bursts_iq]))
    noise_power = mean_burst_power / embed_snr_margin
    noise_std = float(np.sqrt(noise_power / 2.0))

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, n_samples) + 1j * rng.normal(0, noise_std, n_samples)
    iq = noise.astype(np.complex64)

    per_burst_meta = []
    cursor = 0
    for i, (burst_iq, burst_len) in enumerate(zip(bursts_iq, burst_lens)):
        gap = gap_list[i] if gap_list is not None else int(rng.integers(min_burst_gap, max_burst_gap + 1))
        start = cursor + gap
        end = start + burst_len
        if end > n_samples:
            raise ValueError(
                f"Not enough room for burst {i}: would need samples up to {end}, "
                f"but n_samples={n_samples}. Reduce num_bursts/burst_gap or increase n_samples."
            )
        iq[start:end] += burst_iq
        per_burst_meta.append({
            "burst_id": i,
            "true_start": start,
            "true_end": end,
            "burst_len": burst_len,
            "gap_before_burst": gap,
            "burst_power": float(np.mean(np.abs(burst_iq) ** 2)),
        })
        cursor = end

    print(f"[radioml] embedded {len(samples)} bursts in {n_samples}-sample stream "
          f"(mean_burst_power={mean_burst_power:.3e}, embed_noise_power={noise_power:.3e}, "
          f"margin={embed_snr_margin}, gaps={[m['gap_before_burst'] for m in per_burst_meta]}): "
          f"{[(m['true_start'], m['true_end']) for m in per_burst_meta]}")
    return iq, per_burst_meta
