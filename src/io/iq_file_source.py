"""
Formal, reusable IQ file loader -- reads a real captured (or any raw
binary) IQ file into the same 1-D numpy complex64 stream shape every other
IQ source in this repo produces (src/sensing/iq_source.py:generate_synthetic_iq,
src/sensing/radioml_source.py:embed_sample_in_noise), so it can feed the
exact same real energy_detect -> select_aligned_segments -> apply_awn_preprocess
-> AWNModelAdapter/AttackAdapter/TopKAdapter chain src/utils/pipeline.py
already uses -- NOT scripts/sdr_sensing_to_awn_poc.py's placeholder-AWN
standalone script, which this module has no relationship to.

Three supported on-disk formats (`iq_format`):
  - "complex64"           -- raw interleaved real/imag float32 pairs,
                              numpy's native complex64 memory layout
                              (matches src/sensing/iq_source.py's existing
                              GNU-Radio-File-Sink convention).
  - "interleaved_float32" -- raw I,Q,I,Q,... float32 values (two float32
                              per sample, not already packed as complex64).
  - "interleaved_int16"   -- raw I,Q,I,Q,... int16 values, dequantized via
                              `complex_value = (I + jQ) * scale`.

Every format is converted to the SAME output contract: a 1-D np.complex64
array plus a provenance dict recording exactly how it was produced (path,
file size, SHA256, format, endianness, scale, sample_rate, offset, loaded
sample count, dtype, channel count) -- never silently applied without being
recorded.

Does not modify external/AWN, external/adversarial-rf, or the existing
src/sensing/iq_source.py:load_iq_from_file (dead code from the real
pipeline's perspective, per docs/DEPLOYMENT_READINESS.md -- left exactly as
it is; this module is the new, actually-wired-in replacement path).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

SUPPORTED_IQ_FORMATS = ("complex64", "interleaved_float32", "interleaved_int16")
SUPPORTED_ENDIANNESS = ("little", "big", "native")
SUPPORTED_CHANNEL_COUNTS = (1,)  # single complex IQ stream only -- see module docstring

# int16 full-scale dequantization default -- documented, never silently
# applied without being recorded in the returned provenance dict's own
# "scale" field (callers must be able to see this value even if they never
# passed --scale themselves).
DEFAULT_INT16_SCALE = 1.0 / 32768.0


@dataclass
class IQFileProvenance:
    input_path: str
    file_size_bytes: int
    file_sha256: str
    iq_format: str
    endianness: str
    scale: Optional[float]
    sample_rate: Optional[float]
    offset_samples: int
    loaded_sample_count: int
    dtype: str
    channel_count: int

    def to_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "file_size_bytes": self.file_size_bytes,
            "file_sha256": self.file_sha256,
            "iq_format": self.iq_format,
            "endianness": self.endianness,
            "scale": self.scale,
            "sample_rate": self.sample_rate,
            "offset_samples": self.offset_samples,
            "loaded_sample_count": self.loaded_sample_count,
            "dtype": self.dtype,
            "channel_count": self.channel_count,
        }


def _endian_dtype(base: str, endianness: str) -> np.dtype:
    """base: 'f4' (float32), 'i2' (int16), 'c8' (complex64). endianness:
    'little'/'big'/'native' -> numpy dtype byte-order character."""
    order = {"little": "<", "big": ">", "native": "="}[endianness]
    return np.dtype(order + base)


def load_iq_file(
    path: str,
    iq_format: str,
    endianness: str = "native",
    scale: Optional[float] = None,
    sample_rate: Optional[float] = None,
    offset_samples: int = 0,
    max_samples: Optional[int] = None,
    channel_count: int = 1,
) -> tuple[np.ndarray, dict]:
    """Loads `path` as `iq_format`, returns (iq: np.complex64[N], provenance: dict).

    Raises ValueError/FileNotFoundError immediately (never a silent
    fallback to another source or another format) for: missing file, empty
    file, odd interleaved-value count, unsupported iq_format/endianness/
    channel_count, offset_samples < 0, max_samples <= 0 (if given), or
    offset_samples beyond the file's own sample count.
    """
    if iq_format not in SUPPORTED_IQ_FORMATS:
        raise ValueError(f"iq_format must be one of {SUPPORTED_IQ_FORMATS}, got {iq_format!r}")
    if endianness not in SUPPORTED_ENDIANNESS:
        raise ValueError(f"endianness must be one of {SUPPORTED_ENDIANNESS}, got {endianness!r}")
    if channel_count not in SUPPORTED_CHANNEL_COUNTS:
        raise ValueError(
            f"channel_count must be one of {SUPPORTED_CHANNEL_COUNTS} (multi-channel IQ files are "
            f"not supported by this loader), got {channel_count!r}"
        )
    if offset_samples < 0:
        raise ValueError(f"offset_samples must be >= 0, got {offset_samples}")
    if max_samples is not None and max_samples <= 0:
        raise ValueError(f"max_samples must be > 0 if given, got {max_samples}")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"IQ file not found: {path}")
    file_size_bytes = p.stat().st_size
    if file_size_bytes == 0:
        raise ValueError(f"IQ file is empty: {path}")

    file_sha256 = hashlib.sha256(p.read_bytes()).hexdigest()

    if iq_format == "complex64":
        dtype = _endian_dtype("c8", endianness)
        raw = np.fromfile(path, dtype=dtype)
        if raw.size == 0:
            raise ValueError(f"IQ file parsed to 0 complex64 samples (file too short for even one): {path}")
        total_available = raw.size
        if offset_samples > total_available:
            raise ValueError(
                f"offset_samples={offset_samples} exceeds available sample count {total_available} in {path}"
            )
        end = total_available if max_samples is None else min(total_available, offset_samples + max_samples)
        iq = raw[offset_samples:end].astype(np.complex64)
        used_scale = None

    elif iq_format == "interleaved_float32":
        dtype = _endian_dtype("f4", endianness)
        raw = np.fromfile(path, dtype=dtype)
        if raw.size == 0:
            raise ValueError(f"IQ file parsed to 0 float32 values: {path}")
        if raw.size % 2 != 0:
            raise ValueError(
                f"interleaved_float32 requires an EVEN number of float32 values (I,Q,I,Q,...), "
                f"got {raw.size} (odd) in {path}"
            )
        total_available = raw.size // 2
        if offset_samples > total_available:
            raise ValueError(
                f"offset_samples={offset_samples} exceeds available sample count {total_available} in {path}"
            )
        end = total_available if max_samples is None else min(total_available, offset_samples + max_samples)
        i_vals = raw[0::2][offset_samples:end]
        q_vals = raw[1::2][offset_samples:end]
        iq = (i_vals.astype(np.float32) + 1j * q_vals.astype(np.float32)).astype(np.complex64)
        used_scale = None

    else:  # "interleaved_int16"
        dtype = _endian_dtype("i2", endianness)
        raw = np.fromfile(path, dtype=dtype)
        if raw.size == 0:
            raise ValueError(f"IQ file parsed to 0 int16 values: {path}")
        if raw.size % 2 != 0:
            raise ValueError(
                f"interleaved_int16 requires an EVEN number of int16 values (I,Q,I,Q,...), "
                f"got {raw.size} (odd) in {path}"
            )
        total_available = raw.size // 2
        if offset_samples > total_available:
            raise ValueError(
                f"offset_samples={offset_samples} exceeds available sample count {total_available} in {path}"
            )
        end = total_available if max_samples is None else min(total_available, offset_samples + max_samples)
        used_scale = DEFAULT_INT16_SCALE if scale is None else scale
        i_vals = raw[0::2][offset_samples:end].astype(np.float32) * used_scale
        q_vals = raw[1::2][offset_samples:end].astype(np.float32) * used_scale
        iq = (i_vals + 1j * q_vals).astype(np.complex64)

    if iq.size == 0:
        raise ValueError(
            f"Requested slice [offset_samples={offset_samples}:{'end' if max_samples is None else offset_samples + max_samples}] "
            f"produced 0 samples from {path}"
        )
    if not np.isfinite(iq).all():
        raise ValueError(f"IQ file {path} decodes to non-finite (NaN/Inf) values -- refusing to return corrupt data")

    provenance = IQFileProvenance(
        input_path=str(path),
        file_size_bytes=file_size_bytes,
        file_sha256=file_sha256,
        iq_format=iq_format,
        endianness=endianness,
        scale=used_scale,
        sample_rate=sample_rate,
        offset_samples=offset_samples,
        loaded_sample_count=int(iq.size),
        dtype=str(iq.dtype),
        channel_count=channel_count,
    )
    print(f"[iq_file_source] loaded {iq.size} samples from {path} "
          f"(format={iq_format}, endianness={endianness}, scale={used_scale}, "
          f"sha256={file_sha256[:12]}...)")
    return iq, provenance.to_dict()


def write_iq_file(iq: np.ndarray, path: str, iq_format: str, endianness: str = "native",
                   scale: Optional[float] = None) -> None:
    """Inverse of load_iq_file -- writes a complex64 array to disk in the
    given format. Test-fixture/round-trip helper only (not used by the
    formal pipeline itself); used by experiments/test_iq_file_source.py to
    generate the three equivalent test files from one source stream."""
    if iq_format not in SUPPORTED_IQ_FORMATS:
        raise ValueError(f"iq_format must be one of {SUPPORTED_IQ_FORMATS}, got {iq_format!r}")
    if endianness not in SUPPORTED_ENDIANNESS:
        raise ValueError(f"endianness must be one of {SUPPORTED_ENDIANNESS}, got {endianness!r}")

    if iq_format == "complex64":
        dtype = _endian_dtype("c8", endianness)
        iq.astype(dtype).tofile(path)
    elif iq_format == "interleaved_float32":
        dtype = _endian_dtype("f4", endianness)
        interleaved = np.empty(iq.size * 2, dtype=np.float32)
        interleaved[0::2] = iq.real
        interleaved[1::2] = iq.imag
        interleaved.astype(dtype).tofile(path)
    else:  # interleaved_int16
        used_scale = DEFAULT_INT16_SCALE if scale is None else scale
        dtype = _endian_dtype("i2", endianness)
        interleaved = np.empty(iq.size * 2, dtype=np.float64)
        interleaved[0::2] = iq.real / used_scale
        interleaved[1::2] = iq.imag / used_scale
        clipped = np.clip(np.round(interleaved), -32768, 32767)
        clipped.astype(dtype).tofile(path)
